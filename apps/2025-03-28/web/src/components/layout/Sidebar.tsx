'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Box,
  CloseButton,
  Flex,
  Icon,
  useColorModeValue,
  Text,
  BoxProps,
  FlexProps,
} from '@chakra-ui/react'
import {
  FiHome,
  FiTrendingUp,
  FiDollarSign,
  FiCompass,
  FiSettings,
  FiUser,
} from 'react-icons/fi'
import { IconType } from 'react-icons'

interface LinkItemProps {
  name: string
  icon: IconType
  path: string
}

const LinkItems: Array<LinkItemProps> = [
  { name: 'Dashboard', icon: FiHome, path: '/dashboard' },
  { name: 'Portfolio', icon: FiTrendingUp, path: '/portfolio' },
  { name: 'Investments', icon: FiDollarSign, path: '/investments' },
  { name: 'Recommendations', icon: FiCompass, path: '/recommendations' },
  { name: 'Profile', icon: FiUser, path: '/profile' },
  { name: 'Settings', icon: FiSettings, path: '/settings' },
]

interface SidebarProps extends BoxProps {
  onClose: () => void
}

export default function Sidebar({ onClose, ...rest }: SidebarProps) {
  const pathname = usePathname()

  return (
    <Box
      bg={useColorModeValue('white', 'gray.900')}
      borderRight="1px"
      borderRightColor={useColorModeValue('gray.200', 'gray.700')}
      w={{ base: 'full', md: 60 }}
      pos="fixed"
      h="full"
      {...rest}>
      <Flex h="20" alignItems="center" mx="8" justifyContent="space-between">
        <Text fontSize="2xl" fontWeight="bold">
          InvestDash
        </Text>
        <CloseButton display={{ base: 'flex', md: 'none' }} onClick={onClose} />
      </Flex>
      {LinkItems.map((link) => (
        <NavItem
          key={link.name}
          icon={link.icon}
          path={link.path}
          isActive={pathname === link.path}>
          {link.name}
        </NavItem>
      ))}
    </Box>
  )
}

interface NavItemProps extends FlexProps {
  icon: IconType
  path: string
  isActive?: boolean
  children: React.ReactNode
}

const NavItem = ({ icon, path, isActive, children, ...rest }: NavItemProps) => {
  return (
    <Link href={path} style={{ textDecoration: 'none' }}>
      <Flex
        align="center"
        p="4"
        mx="4"
        borderRadius="lg"
        role="group"
        cursor="pointer"
        bg={isActive ? 'blue.400' : 'transparent'}
        color={isActive ? 'white' : 'inherit'}
        _hover={{
          bg: 'blue.400',
          color: 'white',
        }}
        {...rest}>
        {icon && (
          <Icon
            mr="4"
            fontSize="16"
            _groupHover={{
              color: 'white',
            }}
            as={icon}
          />
        )}
        {children}
      </Flex>
    </Link>
  )
} 